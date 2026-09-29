"""
Tests for ``cost.loglogistic_goodput`` and ``loglogistic_shape``
(``machina.swapc.goodput``).

Sections
--------
TestLoglogisticValues    -- G(tau) = 0.5, the closed form, monotone decay
TestLogisticInLogTime    -- equals cost.sigmoid_goodput(k=q, t50=0) at log(L/tau)
TestLoglogisticNumerics  -- relative precision and gradient over L/tau in [1e-8, 1e8],
                            L = 0, L < 0, a huge and an infinite L
TestLoglogisticShape     -- the helper hits G = 0.9 and G = 0.1, extreme latencies
TestLoglogisticArguments -- invalid q, tau and helper latencies are refused

References are computed with ``decimal`` at 50 digits from the exact binary values
of the inputs, so the tolerances are relative (``atol=0``) even where G ~ 1e-80.
"""

import math
from decimal import Decimal, localcontext

import casadi as ca
import numpy as np
import pytest

import machina.swapc  # noqa: F401
from machina.library import registry
from machina.swapc.goodput import loglogistic_shape

pytestmark = pytest.mark.requires_casadi

# L / tau from 1e-8 to 1e8 in quarter decades; 1.0 (L = tau, z = 0) is exactly in the list.
RATIOS = [10.0 ** (k / 4) for k in range(-32, 33)]
QS = [0.5, 1.0, 2.0, 10.0]


def make_goodput(q=2.0, tau=1800.0):
    return registry.get('cost.loglogistic_goodput')(q=q, tau=tau)


def value(descriptor, latency):
    return float(descriptor.function(latency))


def gradient(descriptor, latency):
    x = ca.SX.sym('x')
    g = ca.Function('g', [x], [ca.gradient(descriptor.function(x), x)])
    return float(g(latency))


def exact(latency, q, tau):
    """1 / (1 + (L / tau)**q) to 50 digits."""
    with localcontext() as ctx:
        ctx.prec = 50
        ell = Decimal(latency) / Decimal(tau)
        return float(1 / (1 + ell ** Decimal(q)))


def exact_gradient(latency, q, tau):
    """dG/dL = -q ell**(q - 1) / (tau (1 + ell**q)**2) to 50 digits, ell = L / tau."""
    with localcontext() as ctx:
        ctx.prec = 50
        ell, qd = Decimal(latency) / Decimal(tau), Decimal(q)
        return float(-qd * ell ** (qd - 1) / (Decimal(tau) * (1 + ell ** qd) ** 2))


def in_log_space(latency, q, tau):
    """
    1 / (1 + exp(z)) to 50 digits for z = q * log(L / tau) as CasADi rounds it.

    The factory is handed this z; z itself carries a rounding of order |z| * 1e-16
    that no formula in z can remove (CasADi's log also differs from math.log by an
    ulp for ~0.03 % of arguments). Comparing at the same z measures the formula.
    """
    x = ca.SX.sym('x')
    z = float(ca.Function('z', [x], [q * ca.log(x / tau)])(latency))
    with localcontext() as ctx:
        ctx.prec = 50
        return float(1 / (1 + Decimal(z).exp()))


class TestLoglogisticValues:

    @pytest.mark.parametrize("q", QS + [3.0])
    def test_exactly_half_at_tau(self, q):
        assert value(make_goodput(q=q, tau=1800.0), 1800.0) == 0.5

    @pytest.mark.parametrize("q", QS)
    def test_matches_the_closed_form(self, q):
        """
        Against the exact 1 / (1 + (L / tau)**q). The bound is 1e-13, not 1e-14: z
        is rounded before the formula sees it (|z| up to 184 here).
        """
        tau = 1800.0
        G = make_goodput(q=q, tau=tau)
        for ratio in RATIOS:
            latency = ratio * tau
            np.testing.assert_allclose(value(G, latency), exact(latency, q, tau),
                                       rtol=1e-13, atol=0, err_msg=f"L / tau = {ratio}")

    def test_monotone_decreasing(self):
        G = make_goodput()
        latencies = np.geomspace(1e-3, 1e7, 60)
        values = [value(G, latency) for latency in latencies]
        assert all(b < a for a, b in zip(values, values[1:]))

    def test_io_names(self):
        f = make_goodput().function
        assert f.name_in() == ['latency']
        assert f.name_out() == ['goodput']

    def test_returns_mx_when_called_with_mx(self):
        latency = ca.MX.sym('latency')
        assert isinstance(make_goodput()(latency=latency), ca.MX)


class TestLogisticInLogTime:
    """The log-logistic in L is the logistic in log(L / tau): the same k, t50 = 0."""

    @pytest.mark.parametrize("q", [1.0, 2.0, 5.0])
    def test_equals_sigmoid_in_log_latency(self, q):
        tau = 900.0
        G = make_goodput(q=q, tau=tau)
        sigmoid = registry.get('cost.sigmoid_goodput')(k=q, t50=0.0)
        for latency in (10.0, 450.0, 900.0, 2700.0, 36000.0):
            np.testing.assert_allclose(value(G, latency),
                                       float(sigmoid.function(math.log(latency / tau))),
                                       rtol=1e-12, atol=0)


class TestLoglogisticNumerics:

    @pytest.mark.parametrize("q", QS)
    def test_relative_precision_in_log_space(self, q):
        """0.5 - 0.5 tanh(z / 2) was 8e-4 off at q = 2, L = 1e7 tau and 0 at q = 10, L = 48 tau."""
        tau = 1800.0
        G = make_goodput(q=q, tau=tau)
        for ratio in RATIOS:
            latency = ratio * tau
            np.testing.assert_allclose(value(G, latency), in_log_space(latency, q, tau),
                                       rtol=1e-14, atol=0, err_msg=f"L / tau = {ratio}")

    @pytest.mark.parametrize("q", QS)
    def test_gradient_matches_the_closed_form(self, q):
        """Relative, so the far tail counts; L = tau is z = 0, where max(z, 0) ties."""
        tau = 1800.0
        G = make_goodput(q=q, tau=tau)
        for ratio in RATIOS:
            latency = ratio * tau
            got = gradient(G, latency)
            assert math.isfinite(got)
            np.testing.assert_allclose(got, exact_gradient(latency, q, tau),
                                       rtol=1e-10, atol=0, err_msg=f"L / tau = {ratio}")

    @pytest.mark.parametrize("q", [1.0, 2.0, 10.0])
    def test_one_at_zero_latency_in_the_thesis_range(self, q):
        assert value(make_goodput(q=q), 0.0) >= 1.0 - 1e-15

    @pytest.mark.parametrize("q", [0.25, 0.5])
    def test_within_1e_8_of_one_at_zero_latency_for_q_from_a_quarter(self, q):
        G0 = value(make_goodput(q=q), 0.0)
        assert 1.0 - 1e-8 <= G0 <= 1.0

    @pytest.mark.parametrize("q", [0.5, 2.0, 10.0])
    def test_finite_value_and_gradient_at_zero(self, q):
        G = make_goodput(q=q)
        assert math.isfinite(value(G, 0.0))
        assert gradient(G, 0.0) == 0.0

    @pytest.mark.parametrize("q", [0.5, 2.0])
    def test_a_negative_latency_reads_as_zero(self, q):
        G = make_goodput(q=q)
        assert value(G, -60.0) == value(G, 0.0)
        assert gradient(G, -60.0) == 0.0

    @pytest.mark.parametrize("latency", [1e12, 1e100, 1e300])
    def test_finite_gradient_at_a_huge_latency(self, latency):
        G = make_goodput(q=10.0)
        assert value(G, latency) >= 0.0
        assert math.isfinite(gradient(G, latency))

    @pytest.mark.parametrize("q", [0.5, 10.0])
    def test_zero_at_an_infinite_latency(self, q):
        G = make_goodput(q=q)
        assert value(G, math.inf) == 0.0
        assert gradient(G, math.inf) == 0.0


class TestLoglogisticShape:

    def test_hits_ninety_and_ten_percent(self):
        tau, q = loglogistic_shape(latency_at_90=600.0, latency_at_10=5400.0)
        G = make_goodput(q=q, tau=tau)
        np.testing.assert_allclose(value(G, 600.0), 0.9, rtol=1e-12, atol=0)
        np.testing.assert_allclose(value(G, 5400.0), 0.1, rtol=1e-12, atol=0)

    def test_tau_is_the_geometric_mean(self):
        tau, q = loglogistic_shape(latency_at_90=600.0, latency_at_10=5400.0)
        assert tau == 1800.0  # sqrt(3240000) is exact; sqrt(600) * sqrt(5400) is not
        np.testing.assert_allclose(q, 2.0, rtol=1e-15)

    @pytest.mark.parametrize("a, b", [(1e200, 1e201), (1e-200, 1e-190)])
    def test_extreme_latencies_give_a_finite_tau(self, a, b):
        """a * b overflows (underflows) here; sqrt(a) * sqrt(b) does not."""
        tau, q = loglogistic_shape(latency_at_90=a, latency_at_10=b)
        assert math.isfinite(tau) and tau > 0.0
        # (a / tau)**q = 1/9 and (b / tau)**q = 9, in log space.
        np.testing.assert_allclose(q * (math.log(a) - math.log(tau)), -math.log(9.0),
                                   rtol=1e-12, atol=0)
        np.testing.assert_allclose(q * (math.log(b) - math.log(tau)), math.log(9.0),
                                   rtol=1e-12, atol=0)
        G = make_goodput(q=q, tau=tau)
        np.testing.assert_allclose(value(G, a), 0.9, rtol=1e-12, atol=0)
        np.testing.assert_allclose(value(G, b), 0.1, rtol=1e-12, atol=0)

    def test_a_ratio_beyond_the_float_range_gives_a_finite_q(self):
        """b / a overflows to inf; q then comes from log(b) - log(a)."""
        tau, q = loglogistic_shape(latency_at_90=1e-300, latency_at_10=1e300)
        np.testing.assert_allclose(tau, 1.0, rtol=1e-15)
        np.testing.assert_allclose(q, math.log(81.0) / (600.0 * math.log(10.0)), rtol=1e-14)


class TestLoglogisticArguments:

    @pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan, True, "2"])
    def test_q_refused(self, bad):
        with pytest.raises(ValueError, match="cost.loglogistic_goodput: q"):
            make_goodput(q=bad)

    @pytest.mark.parametrize("bad", [0.0, -60.0, math.inf, None])
    def test_tau_refused(self, bad):
        with pytest.raises(ValueError, match="cost.loglogistic_goodput: tau"):
            make_goodput(tau=bad)

    def test_shape_needs_ordered_latencies(self):
        with pytest.raises(ValueError, match="latency_at_90 .* must be smaller"):
            loglogistic_shape(latency_at_90=600.0, latency_at_10=600.0)

    def test_shape_needs_positive_latencies(self):
        with pytest.raises(ValueError, match="loglogistic_shape: latency_at_90"):
            loglogistic_shape(latency_at_90=0.0, latency_at_10=600.0)
