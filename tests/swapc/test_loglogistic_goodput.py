"""
Tests for ``cost.loglogistic_goodput`` and ``loglogistic_shape``
(``machina.swapc.goodput``).

Sections
--------
TestLoglogisticValues    -- G(tau) = 0.5, the closed form, monotone decay
TestLogisticInLogTime    -- equals cost.sigmoid_goodput(k=q, t50=0) at log(L/tau)
TestLoglogisticNumerics  -- finite value and gradient at L = 0 and at a huge L
TestLoglogisticShape     -- the helper hits G = 0.9 and G = 0.1
TestLoglogisticArguments -- invalid q, tau and helper latencies are refused
"""

import math

import casadi as ca
import numpy as np
import pytest

import machina.swapc  # noqa: F401
from machina.library import registry
from machina.swapc.goodput import loglogistic_shape

pytestmark = pytest.mark.requires_casadi


def make_goodput(q=2.0, tau=1800.0):
    return registry.get('cost.loglogistic_goodput')(q=q, tau=tau)


def value(descriptor, latency):
    return float(descriptor.function(latency))


def gradient(descriptor, latency):
    x = ca.SX.sym('x')
    g = ca.Function('g', [x], [ca.gradient(descriptor.function(x), x)])
    return float(g(latency))


def closed_form(latency, q, tau):
    return 1.0 / (1.0 + (latency / tau) ** q)


class TestLoglogisticValues:

    def test_half_at_tau(self):
        np.testing.assert_allclose(value(make_goodput(q=3.0, tau=1800.0), 1800.0), 0.5,
                                   rtol=0, atol=1e-15)

    def test_one_at_zero_latency(self):
        np.testing.assert_allclose(value(make_goodput(), 0.0), 1.0, rtol=0, atol=1e-15)

    @pytest.mark.parametrize("q", [0.5, 1.0, 2.0, 10.0])
    def test_matches_the_closed_form(self, q):
        G = make_goodput(q=q, tau=1800.0)
        for latency in (1.0, 60.0, 300.0, 1800.0, 7200.0, 86400.0):
            np.testing.assert_allclose(value(G, latency), closed_form(latency, q, 1800.0),
                                       rtol=1e-12, atol=1e-15)

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
                                       rtol=1e-12, atol=1e-15)


class TestLoglogisticNumerics:

    @pytest.mark.parametrize("q", [0.5, 2.0, 10.0])
    def test_finite_value_and_gradient_at_zero(self, q):
        G = make_goodput(q=q)
        assert math.isfinite(value(G, 0.0))
        assert gradient(G, 0.0) == 0.0

    @pytest.mark.parametrize("latency", [1e12, 1e100, 1e300])
    def test_finite_gradient_at_a_huge_latency(self, latency):
        G = make_goodput(q=10.0)
        assert value(G, latency) >= 0.0
        assert math.isfinite(gradient(G, latency))

    def test_gradient_matches_the_closed_form(self):
        q, tau, latency = 2.0, 1800.0, 900.0
        ell = latency / tau
        expected = -q * ell ** (q - 1.0) / tau / (1.0 + ell ** q) ** 2
        np.testing.assert_allclose(gradient(make_goodput(q=q, tau=tau), latency), expected,
                                   rtol=1e-10, atol=0)


class TestLoglogisticShape:

    def test_hits_ninety_and_ten_percent(self):
        tau, q = loglogistic_shape(latency_at_90=600.0, latency_at_10=5400.0)
        G = make_goodput(q=q, tau=tau)
        np.testing.assert_allclose(value(G, 600.0), 0.9, rtol=0, atol=1e-12)
        np.testing.assert_allclose(value(G, 5400.0), 0.1, rtol=0, atol=1e-12)

    def test_tau_is_the_geometric_mean(self):
        tau, q = loglogistic_shape(latency_at_90=600.0, latency_at_10=5400.0)
        np.testing.assert_allclose(tau, 1800.0, rtol=1e-15)
        np.testing.assert_allclose(q, 2.0, rtol=1e-15)


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
