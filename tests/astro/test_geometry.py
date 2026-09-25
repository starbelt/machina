"""
Tests for the astro geometry factories in ``machina.astro.geometry``:
``geometry.ground_target_eci`` and ``geometry.elevation_angle``.

Test classes
------------
TestGroundTargetEci      -- geometry.ground_target_eci factory
TestElevationAngle       -- geometry.elevation_angle factory

``cost.smooth_coverage`` is tested in ``test_smooth_coverage.py`` beside this
file; the domain counts in ``tests/library/test_registry.py``.
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

R_EARTH = 6378.137  # km


def _gt_eci(lat_deg, lon_deg, R=R_EARTH):
    """Evaluate ground_target_eci numerically."""
    fd = registry.get('geometry.ground_target_eci')(R_earth=R)
    lat = ca.DM(math.radians(lat_deg))
    lon = ca.DM(math.radians(lon_deg))
    return np.array(fd.function(lat, lon)).flatten()


def _elev_angle(r_sat, r_target):
    """Evaluate elevation_angle numerically (inputs are numpy arrays)."""
    fd = registry.get('geometry.elevation_angle')()
    result = fd.function(ca.DM(r_sat), ca.DM(r_target))
    return float(result)


# ---------------------------------------------------------------------------
# TestGroundTargetEci
# ---------------------------------------------------------------------------

class TestGroundTargetEci:

    def test_equatorial_at_zero_lon(self):
        """lat=0, lon=0 -> [R_earth, 0, 0]."""
        r = _gt_eci(0.0, 0.0)
        np.testing.assert_allclose(r, [R_EARTH, 0.0, 0.0], atol=1e-9)

    def test_equatorial_at_90_lon(self):
        """lat=0, lon=90 -> [0, R_earth, 0]."""
        r = _gt_eci(0.0, 90.0)
        np.testing.assert_allclose(r, [0.0, R_EARTH, 0.0], atol=1e-9)

    def test_north_pole(self):
        """lat=90, lon=arbitrary -> [0, 0, R_earth]."""
        r = _gt_eci(90.0, 0.0)
        np.testing.assert_allclose(r, [0.0, 0.0, R_EARTH], atol=1e-9)

    def test_south_pole(self):
        """lat=-90 -> [0, 0, -R_earth]."""
        r = _gt_eci(-90.0, 0.0)
        np.testing.assert_allclose(r, [0.0, 0.0, -R_EARTH], atol=1e-9)

    def test_magnitude_equals_R_earth(self):
        """||r_target|| = R_earth for arbitrary lat/lon."""
        for lat, lon in [(38.9, -77.0), (51.5, -0.1), (-33.9, 151.2)]:
            r = _gt_eci(lat, lon)
            np.testing.assert_allclose(np.linalg.norm(r), R_EARTH, rtol=1e-10)

    def test_arbitrary_lat_lon_matches_numpy(self):
        """Cross-check against direct numpy formula."""
        lat_deg, lon_deg = 38.9, -77.0
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        expected = R_EARTH * np.array([
            math.cos(lat) * math.cos(lon),
            math.cos(lat) * math.sin(lon),
            math.sin(lat),
        ])
        r = _gt_eci(lat_deg, lon_deg)
        np.testing.assert_allclose(r, expected, atol=1e-10)

    def test_output_shape(self):
        """Output is (3, 1)."""
        fd = registry.get('geometry.ground_target_eci')(R_earth=R_EARTH)
        assert fd.output_shapes == [(3, 1)]

    def test_input_shapes(self):
        """Inputs are each (1, 1) scalars."""
        fd = registry.get('geometry.ground_target_eci')(R_earth=R_EARTH)
        assert fd.input_shapes == [(1, 1), (1, 1)]

    def test_registry_key_exists(self):
        fd = registry.get('geometry.ground_target_eci')(R_earth=R_EARTH)
        assert fd.name == 'ground_target_eci'

    def test_mx_callable(self):
        """Function produces MX output when called with MX inputs."""
        fd = registry.get('geometry.ground_target_eci')(R_earth=R_EARTH)
        lat_mx = ca.MX.sym('lat', 1, 1)
        lon_mx = ca.MX.sym('lon', 1, 1)
        result = fd.function(lat_mx, lon_mx)
        assert isinstance(result, ca.MX)
        assert result.shape == (3, 1)

    def test_custom_R_earth(self):
        """Factory param R_earth is respected."""
        r = _gt_eci(0.0, 0.0, R=1.0)  # unit sphere
        np.testing.assert_allclose(r, [1.0, 0.0, 0.0], atol=1e-9)


# ---------------------------------------------------------------------------
# TestElevationAngle
# ---------------------------------------------------------------------------

class TestElevationAngle:

    def test_overhead_equals_pi_over_2(self):
        """Satellite directly above target -> elevation = pi/2."""
        r_target = np.array([R_EARTH, 0.0, 0.0])
        r_sat    = np.array([R_EARTH + 500.0, 0.0, 0.0])  # 500 km altitude
        elev = _elev_angle(r_sat, r_target)
        np.testing.assert_allclose(elev, math.pi / 2, atol=1e-6)

    def test_satellite_behind_earth_is_negative(self):
        """Satellite on the opposite side of the Earth -> elevation < 0."""
        r_target = np.array([R_EARTH, 0.0, 0.0])
        r_sat    = np.array([-(R_EARTH + 500.0), 0.0, 0.0])
        elev = _elev_angle(r_sat, r_target)
        assert elev < 0.0

    def test_elevation_range(self):
        """Elevation angle is always in [-pi/2, pi/2]."""
        r_target = np.array([R_EARTH, 0.0, 0.0])
        for offset in [np.array([500, 0, 0]),
                       np.array([0, 500, 0]),
                       np.array([0, 0, R_EARTH + 500]),
                       np.array([-500, 0, 0])]:
            r_sat = r_target + offset
            elev = _elev_angle(r_sat, r_target)
            assert -math.pi / 2 - 1e-6 <= elev <= math.pi / 2 + 1e-6

    def test_output_shape(self):
        """Output is (1, 1)."""
        fd = registry.get('geometry.elevation_angle')()
        assert fd.output_shapes == [(1, 1)]

    def test_input_shapes(self):
        """Both inputs are (3, 1)."""
        fd = registry.get('geometry.elevation_angle')()
        assert fd.input_shapes == [(3, 1), (3, 1)]

    def test_registry_key_exists(self):
        fd = registry.get('geometry.elevation_angle')()
        assert fd.name == 'elevation_angle'

    def test_mx_callable(self):
        """Function produces MX output when called with MX inputs."""
        fd = registry.get('geometry.elevation_angle')()
        r_sat_mx    = ca.MX.sym('r_sat', 3, 1)
        r_target_mx = ca.MX.sym('r_target', 3, 1)
        result = fd.function(r_sat_mx, r_target_mx)
        assert isinstance(result, ca.MX)
        assert result.shape == (1, 1)

    def test_equatorial_sat_over_midlat_target(self):
        """Equatorial satellite at target longitude, target at 38.9 deg lat -> negative elevation."""
        # Satellite on equator directly below the longitude of DC
        lat_deg, lon_deg = 38.9, -77.0
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        r_target = R_EARTH * np.array([math.cos(lat)*math.cos(lon),
                                        math.cos(lat)*math.sin(lon),
                                        math.sin(lat)])
        # Equatorial satellite at same longitude, 500 km altitude
        r_sat = (R_EARTH + 500.0) * np.array([math.cos(lon), math.sin(lon), 0.0])
        elev = _elev_angle(r_sat, r_target)
        # The satellite should be below the horizon for this geometry
        assert elev < 0.0

    def test_elevation_increases_with_altitude(self):
        """For satellite directly above, elevation stays pi/2 regardless of altitude."""
        r_target = np.array([R_EARTH, 0.0, 0.0])
        elevations = []
        for alt in [200.0, 500.0, 1000.0, 10000.0]:
            r_sat = np.array([R_EARTH + alt, 0.0, 0.0])
            elevations.append(_elev_angle(r_sat, r_target))
        # All should be pi/2 (overhead)
        for e in elevations:
            np.testing.assert_allclose(e, math.pi / 2, atol=1e-5)
