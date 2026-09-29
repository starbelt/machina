"""
Tests for the formation relative-motion factories in ``machina.astro.relative``:
``transform.roe_to_rtn``, ``geometry.rn_min_separation``, ``util.synodic_period``
and the plain ring helpers.

Test classes
------------
TestRoeToRtn                  -- closed form, shapes, linearity
TestRoeToRtnAgainstTwoBody    -- linear map vs exact two-body positions through
                                 transform.mee_to_eci: O(a |delta|^2) residual
TestRnMinSeparation           -- parallel / perpendicular limits, brute force, gradients
TestSynodicPeriod             -- exact formula, small-dr approximation, equal radii
TestRingHelpers               -- ring_pass_interval, ring_size_for_interval

``geometry.off_axis_angle`` and the cone factories are tested in ``test_beam.py``
beside this file.
"""

import math

import casadi as ca
import numpy as np
import pytest

import machina.astro  # noqa: F401
from machina.astro.relative import ring_pass_interval, ring_size_for_interval
from machina.library import registry

pytestmark = pytest.mark.requires_casadi

MU_EARTH = 398600.4418   # km^3/s^2
A_GEO = 42164.0          # km


# ===========================================================================
# Helpers
# ===========================================================================

def _roe_to_rtn(roe, lam, a=A_GEO):
    fd = registry.get('transform.roe_to_rtn')(a=a)
    return np.array(fd.function(ca.DM(roe), ca.DM(lam))).flatten()


def _closed_form(roe, lam, a=A_GEO):
    """R, T, N written out in numpy, independent of the factory."""
    dlam, df, dg, dix, diy = roe
    c, s = np.cos(lam), np.sin(lam)
    return a * np.array([-(df * c + dg * s),
                         dlam + 2.0 * (df * s - dg * c),
                         dix * s - diy * c])


def _d_min(de, di, a=A_GEO):
    fd = registry.get('geometry.rn_min_separation')(a=a)
    return float(fd.function(ca.DM(de), ca.DM(di)))


def _mee_to_eci(mee):
    fd = registry.get('transform.mee_to_eci')(mu=MU_EARTH)
    r, v = fd.function(ca.DM(mee))
    return np.array(r).flatten(), np.array(v).flatten()


def _true_longitude(mean_longitude, f, g):
    """True longitude from mean longitude by a numpy Newton solve of Kepler's equation."""
    e = math.hypot(f, g)
    varpi = math.atan2(g, f)
    M = mean_longitude - varpi
    E = M
    for _ in range(50):
        E -= (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
    nu = 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(E / 2.0),
                          math.sqrt(1.0 - e) * math.cos(E / 2.0))
    return varpi + nu


def _exact_rtn(roe, lam, a=A_GEO):
    """Deputy position in the chief's RTN from exact two-body positions.

    Chief MEE: p = a, f = g = h = k = 0, L = lam. Deputy MEE: same semi-major axis
    (p = a (1 - e^2)), f = df, g = dg, h = dix / 2, k = diy / 2, true longitude
    from the mean longitude lam + dlam.
    """
    dlam, df, dg, dix, diy = roe
    r_c, v_c = _mee_to_eci([a, 0.0, 0.0, 0.0, 0.0, lam])
    e2 = df ** 2 + dg ** 2
    L_d = _true_longitude(lam + dlam, df, dg)
    r_d, _ = _mee_to_eci([a * (1.0 - e2), df, dg, dix / 2.0, diy / 2.0, L_d])

    R_hat = r_c / np.linalg.norm(r_c)
    N_hat = np.cross(r_c, v_c)
    N_hat /= np.linalg.norm(N_hat)
    T_hat = np.cross(N_hat, R_hat)
    d = r_d - r_c
    return np.array([d @ R_hat, d @ T_hat, d @ N_hat])


def _worst_normalised_residual(unit_roes, scale, longitudes):
    """max over cases and longitudes of |linear - exact| / (a max|delta|^2)."""
    worst = 0.0
    for unit in unit_roes:
        roe = unit * scale
        size = A_GEO * np.max(np.abs(roe)) ** 2
        for lam in longitudes:
            err = np.linalg.norm(_roe_to_rtn(roe, lam) - _exact_rtn(roe, lam))
            worst = max(worst, err / size)
    return worst


def _synodic(a1, a2, mu=MU_EARTH):
    fd = registry.get('util.synodic_period')(mu=mu)
    return float(fd.function(ca.DM(a1), ca.DM(a2)))


# ===========================================================================
# transform.roe_to_rtn
# ===========================================================================

class TestRoeToRtn:

    def test_descriptor_shapes(self):
        fd = registry.get('transform.roe_to_rtn')(a=A_GEO)
        assert fd.name == 'roe_to_rtn'
        assert fd.input_names == ['roe', 'lam']
        assert fd.input_shapes == [(5, 1), (1, 1)]
        assert fd.output_names == ['r_rtn']
        assert fd.output_shapes == [(3, 1)]

    def test_matches_closed_form(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            roe = rng.normal(size=5) * 1e-4
            for lam in np.linspace(0.0, 2.0 * np.pi, 9):
                np.testing.assert_allclose(_roe_to_rtn(roe, lam), _closed_form(roe, lam),
                                           rtol=0.0, atol=1e-12)

    def test_zero_roe_is_the_chief(self):
        np.testing.assert_allclose(_roe_to_rtn(np.zeros(5), 0.7), np.zeros(3), atol=0.0)

    def test_along_track_offset_only(self):
        """dlam alone places the deputy a * dlam ahead, constant in lam."""
        for lam in (0.0, 1.0, 4.0):
            np.testing.assert_allclose(_roe_to_rtn([1e-4, 0, 0, 0, 0], lam),
                                       [0.0, A_GEO * 1e-4, 0.0], atol=1e-12)

    def test_sign_conventions_at_lam_zero(self):
        """At lam = 0: +df puts the deputy below the chief (R < 0, perigee along +x),
        +dg puts it behind (T < 0), +diy puts it south (N < 0)."""
        R, T, N = _roe_to_rtn([0, 1e-4, 0, 0, 0], 0.0)
        assert R < 0 and abs(T) < 1e-12 and abs(N) < 1e-12
        R, T, N = _roe_to_rtn([0, 0, 1e-4, 0, 0], 0.0)
        assert T < 0 and abs(R) < 1e-12
        R, T, N = _roe_to_rtn([0, 0, 0, 0, 1e-4], 0.0)
        assert N < 0

    def test_non_positive_a_refused(self):
        with pytest.raises(ValueError, match="transform.roe_to_rtn"):
            registry.get('transform.roe_to_rtn')(a=0.0)


class TestRoeToRtnAgainstTwoBody:
    """The linear map's residual against exact two-body geometry is O(a |delta|^2).

    Measured: the worst |linear - exact| / (a max|delta|^2) over 12 random
    unit-scale ROE and 12 longitudes is 3.46 at both |delta| = 1e-4 (offsets up to
    ~4 km, residual 1.27 m) and 1e-5 (residual 1.27 cm); other seeds reach ~6.
    C = 10 bounds them.
    """

    C = 10.0
    LONGITUDES = np.linspace(0.0, 2.0 * np.pi, 13)[:-1]

    @staticmethod
    def _unit_roes():
        rng = np.random.default_rng(0)
        return [rng.uniform(-1.0, 1.0, 5) for _ in range(12)]

    @pytest.mark.parametrize('scale', [1e-4, 1e-5])
    def test_residual_is_bounded_by_c_a_delta_squared(self, scale):
        worst = _worst_normalised_residual(self._unit_roes(), scale, self.LONGITUDES)
        assert worst <= self.C, (
            f"|linear - exact| / (a max|delta|^2) = {worst:.3g} at |delta| ~ {scale:g}")

    def test_residual_converges_quadratically(self):
        """Ten times smaller deltas give a ~hundred times smaller residual."""
        unit_roes = self._unit_roes()
        abs_errors = []
        for scale in (1e-4, 1e-5):
            abs_errors.append(max(
                np.linalg.norm(_roe_to_rtn(u * scale, lam) - _exact_rtn(u * scale, lam))
                for u in unit_roes for lam in self.LONGITUDES))
        ratio = abs_errors[0] / abs_errors[1]
        assert 50.0 < ratio < 200.0, f"residual ratio over a 10x delta step was {ratio:.3g}"

    def test_first_order_terms_dominate(self):
        """The linear offset is kilometres while the residual is metres: a wrong
        sign in any row would leave an O(a |delta|) residual and fail here."""
        roe = np.array([2e-5, 1e-4, -6e-5, 8e-5, -3e-5])
        for lam in self.LONGITUDES:
            exact = _exact_rtn(roe, lam)
            linear = _roe_to_rtn(roe, lam)
            np.testing.assert_allclose(linear, exact, rtol=0.0,
                                       atol=self.C * A_GEO * np.max(np.abs(roe)) ** 2)


# ===========================================================================
# geometry.rn_min_separation
# ===========================================================================

class TestRnMinSeparation:

    def test_descriptor_shapes(self):
        fd = registry.get('geometry.rn_min_separation')(a=A_GEO)
        assert fd.input_names == ['de', 'di']
        assert fd.input_shapes == [(2, 1), (2, 1)]
        assert fd.output_shapes == [(1, 1)]

    @pytest.mark.parametrize('phase', [0.0, 0.8, 2.5, -1.9])
    def test_parallel_vectors_give_a_times_the_smaller(self, phase):
        u = np.array([math.cos(phase), math.sin(phase)])
        np.testing.assert_allclose(_d_min(1.2e-4 * u, 0.7e-4 * u), A_GEO * 0.7e-4, rtol=1e-12)
        np.testing.assert_allclose(_d_min(0.5e-4 * u, 0.9e-4 * u), A_GEO * 0.5e-4, rtol=1e-12)

    def test_antiparallel_vectors_give_a_times_the_smaller(self):
        u = np.array([0.6, 0.8])
        np.testing.assert_allclose(_d_min(1e-4 * u, -0.4e-4 * u), A_GEO * 0.4e-4, rtol=1e-12)

    def test_equal_parallel_vectors(self):
        """|de| = |di| parallel: the eigenvalues cross (disc = 0)."""
        u = np.array([1.0, 0.0])
        np.testing.assert_allclose(_d_min(1e-4 * u, 1e-4 * u), A_GEO * 1e-4, rtol=1e-12)

    @pytest.mark.parametrize('phase', [0.0, 1.1, 3.0])
    def test_perpendicular_vectors_give_zero(self, phase):
        u = np.array([math.cos(phase), math.sin(phase)])
        w = np.array([-u[1], u[0]])
        assert _d_min(1e-4 * u, 2e-4 * w) < 1e-9

    def test_zero_eccentricity_vector_gives_zero(self):
        """de = 0: the deputy crosses the chief's orbit plane on the chief's radius."""
        assert _d_min([0.0, 0.0], [1e-4, 3e-5]) < 1e-9

    def test_matches_brute_force_over_longitude(self):
        """Brute force over 10^4 samples of sqrt(R^2 + N^2) from transform.roe_to_rtn.
        The formula is a lower bound; the sampled minimum exceeds it by less than
        the sampling error, ~1e-4 a |delta| at this density."""
        fd = registry.get('transform.roe_to_rtn')(a=A_GEO)
        lam = np.linspace(0.0, 2.0 * np.pi, 10_000, endpoint=False)
        rng = np.random.default_rng(1)
        for _ in range(25):
            de = rng.normal(size=2) * 1e-4
            di = rng.normal(size=2) * 1e-4
            roe = np.r_[rng.normal() * 1e-4, de, di]
            r_rtn = np.array(fd.function.map(lam.size)(ca.DM(roe), ca.DM(lam).T))
            brute = np.sqrt(r_rtn[0] ** 2 + r_rtn[2] ** 2).min()
            formula = _d_min(de, di)
            scale = A_GEO * np.max(np.abs(np.r_[de, di]))
            assert formula <= brute + 1e-9
            np.testing.assert_allclose(formula, brute, rtol=0.0, atol=2e-4 * scale)

    @pytest.mark.parametrize('de, di', [
        ([1e-4, 0.0], [1e-4, 0.0]),        # |de| = |di| parallel: disc = 0
        ([0.0, 0.0], [1e-4, 2e-5]),        # de = 0: lambda_min = 0
        ([1e-4, 0.0], [0.0, 1e-4]),        # perpendicular: lambda_min = 0
        ([0.0, 0.0], [0.0, 0.0]),          # both zero
        ([3e-5, -7e-5], [2e-5, 6e-5]),     # generic
    ])
    def test_gradient_is_finite(self, de, di):
        fd = registry.get('geometry.rn_min_separation')(a=A_GEO)
        x_de = ca.SX.sym('de', 2)
        x_di = ca.SX.sym('di', 2)
        d_min = fd.function(x_de, x_di)
        jac = ca.Function('d_min_jacobian', [x_de, x_di],
                          [ca.jacobian(d_min, ca.vertcat(x_de, x_di))])
        values = np.array(jac(ca.DM(de), ca.DM(di)))
        assert np.all(np.isfinite(values)), f"non-finite gradient {values} at de={de}, di={di}"

    def test_non_positive_a_refused(self):
        with pytest.raises(ValueError, match="geometry.rn_min_separation"):
            registry.get('geometry.rn_min_separation')(a=-1.0)


# ===========================================================================
# util.synodic_period
# ===========================================================================

class TestSynodicPeriod:

    def test_descriptor_shapes(self):
        fd = registry.get('util.synodic_period')(mu=MU_EARTH)
        assert fd.input_names == ['a1', 'a2']
        assert fd.output_names == ['T_syn']

    def test_matches_exact_formula(self):
        for a1, a2 in [(A_GEO, A_GEO + 10.0), (7000.0, 7500.0), (A_GEO + 100.0, A_GEO)]:
            n1 = math.sqrt(MU_EARTH / a1 ** 3)
            n2 = math.sqrt(MU_EARTH / a2 ** 3)
            np.testing.assert_allclose(_synodic(a1, a2), 2.0 * math.pi / abs(n1 - n2),
                                       rtol=1e-12)

    def test_symmetric(self):
        np.testing.assert_allclose(_synodic(A_GEO, A_GEO + 50.0),
                                   _synodic(A_GEO + 50.0, A_GEO), rtol=1e-14)

    @pytest.mark.parametrize('dr', [1.0, 10.0, 100.0, 1000.0])
    def test_small_dr_approximation(self, dr):
        """T_syn = (2/3) T a / dr (1 + 1.25 dr/a + O((dr/a)^2)) for a2 = a + dr."""
        T = 2.0 * math.pi * math.sqrt(A_GEO ** 3 / MU_EARTH)
        approx = (2.0 / 3.0) * T * A_GEO / dr
        x = dr / A_GEO
        rel = _synodic(A_GEO, A_GEO + dr) / approx - 1.0
        assert 0.0 < rel <= 1.5 * x
        np.testing.assert_allclose(rel / x, 1.25, rtol=0.05)

    def test_ten_km_at_geo_is_years(self):
        days = _synodic(A_GEO, A_GEO + 10.0) / 86400.0
        assert 2700.0 < days < 2900.0

    def test_equal_radii_is_never(self):
        assert _synodic(A_GEO, A_GEO) > 1e30

    def test_gradient_finite_at_equal_radii(self):
        fd = registry.get('util.synodic_period')(mu=MU_EARTH)
        a1 = ca.SX.sym('a1')
        a2 = ca.SX.sym('a2')
        T_syn = fd.function(a1, a2)
        jac = ca.Function('t_syn_jacobian', [a1, a2], [ca.jacobian(T_syn, ca.vertcat(a1, a2))])
        assert np.all(np.isfinite(np.array(jac(A_GEO, A_GEO))))

    def test_non_positive_mu_refused(self):
        with pytest.raises(ValueError, match="util.synodic_period"):
            registry.get('util.synodic_period')(mu=0.0)


# ===========================================================================
# Ring helpers
# ===========================================================================

class TestRingHelpers:

    def test_pass_interval(self):
        assert ring_pass_interval(3600.0, 4) == 900.0
        assert ring_pass_interval(100.0, 1) == 100.0

    def test_size_rounds_up(self):
        assert ring_size_for_interval(90.0, 30.0) == 3
        assert ring_size_for_interval(90.001, 30.0) == 4
        assert ring_size_for_interval(100.0, 30.0) == 4
        assert ring_size_for_interval(10.0, 30.0) == 1

    def test_size_and_interval_agree(self):
        T_syn = _synodic(A_GEO, A_GEO + 100.0)
        n = ring_size_for_interval(T_syn, 600.0)
        assert ring_pass_interval(T_syn, n) <= 600.0
        assert ring_pass_interval(T_syn, n - 1) > 600.0

    @pytest.mark.parametrize('T_syn, n', [(0.0, 3), (-1.0, 3), (math.inf, 3), (100.0, 0),
                                          (100.0, -2), (100.0, 2.5), (100.0, True)])
    def test_pass_interval_refuses_bad_input(self, T_syn, n):
        with pytest.raises(ValueError, match="ring_pass_interval"):
            ring_pass_interval(T_syn, n)

    @pytest.mark.parametrize('T_syn, T_target', [(0.0, 1.0), (1.0, 0.0), (-5.0, 1.0),
                                                 (1.0, math.nan)])
    def test_size_refuses_bad_input(self, T_syn, T_target):
        with pytest.raises(ValueError, match="ring_size_for_interval"):
            ring_size_for_interval(T_syn, T_target)
