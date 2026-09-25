"""
Tests for ``cost.smooth_coverage``, the sigmoid coverage indicator in
``machina.astro.coverage``.

Test classes
------------
TestSmoothCoverage       -- cost.smooth_coverage factory
"""

import math

import casadi as ca
import numpy as np
import pytest

import machina.astro  # noqa: F401
from machina.library import registry

pytestmark = pytest.mark.requires_casadi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smooth_cov(elevation, min_elev, k):
    """Evaluate smooth_coverage numerically."""
    fd = registry.get('cost.smooth_coverage')()
    result = fd.function(ca.DM(elevation), ca.DM(min_elev), ca.DM(k))
    return float(result)


# ---------------------------------------------------------------------------
# TestSmoothCoverage
# ---------------------------------------------------------------------------

class TestSmoothCoverage:

    def test_registry_key_exists(self):
        fd = registry.get('cost.smooth_coverage')()
        assert fd.name == 'smooth_coverage'

    def test_at_threshold_equals_half(self):
        """coverage(eps_min, eps_min, k) = 0.5 exactly."""
        for k in [5.0, 20.0, 50.0]:
            eps_min = math.radians(10.0)
            cov = _smooth_cov(eps_min, eps_min, k)
            np.testing.assert_allclose(cov, 0.5, atol=1e-12)

    def test_well_above_threshold_close_to_one(self):
        """coverage >> 0.5 when elevation >> min_elevation."""
        eps_min = math.radians(10.0)
        cov = _smooth_cov(math.radians(60.0), eps_min, 20.0)
        assert cov > 0.99

    def test_well_below_threshold_close_to_zero(self):
        """coverage << 0.5 when elevation << min_elevation."""
        eps_min = math.radians(10.0)
        cov = _smooth_cov(math.radians(-20.0), eps_min, 20.0)
        assert cov < 0.01

    def test_monotonically_increasing(self):
        """Coverage is strictly increasing with elevation."""
        eps_min = math.radians(10.0)
        k = 20.0
        elevations = np.linspace(math.radians(-30), math.radians(60), 20)
        covs = [_smooth_cov(e, eps_min, k) for e in elevations]
        for c1, c2 in zip(covs[:-1], covs[1:]):
            assert c1 < c2

    def test_output_in_unit_interval(self):
        """Coverage is always in (0, 1)."""
        eps_min = math.radians(10.0)
        for elev in np.linspace(math.radians(-90), math.radians(90), 50):
            cov = _smooth_cov(elev, eps_min, 20.0)
            assert 0.0 < cov < 1.0

    def test_output_shape(self):
        """Output is (1, 1)."""
        fd = registry.get('cost.smooth_coverage')()
        assert fd.output_shapes == [(1, 1)]

    def test_input_shapes(self):
        """All three inputs are (1, 1)."""
        fd = registry.get('cost.smooth_coverage')()
        assert fd.input_shapes == [(1, 1), (1, 1), (1, 1)]

    def test_mx_callable(self):
        """Function produces MX output with MX inputs."""
        fd = registry.get('cost.smooth_coverage')()
        e_mx     = ca.MX.sym('elevation', 1, 1)
        emin_mx  = ca.MX.sym('min_elevation', 1, 1)
        k_mx     = ca.MX.sym('k', 1, 1)
        result = fd.function(e_mx, emin_mx, k_mx)
        assert isinstance(result, ca.MX)
        assert result.shape == (1, 1)

    def test_steeper_k_gives_sharper_transition(self):
        """Higher k -> transition is more concentrated near eps_min."""
        eps_min = math.radians(10.0)
        # Slightly above threshold
        eps = math.radians(11.0)
        cov_low_k  = _smooth_cov(eps, eps_min, 5.0)
        cov_high_k = _smooth_cov(eps, eps_min, 50.0)
        # Higher k means faster rise above threshold
        assert cov_high_k > cov_low_k
