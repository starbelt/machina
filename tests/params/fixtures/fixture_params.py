"""
The declaring module behind ``tests/params/fixtures/params.csv``.

A worked example of the pipeline and the table ``make check-params`` checks:
declared here, valued by hand in the CSV with a provenance code and a source,
checked by ``machina params check``. ``tests/params/test_fixture_table.py``
keeps the two in agreement.
"""

from machina.params import param

REFRESH_S = param("REFRESH_S", type="f64", unit="s", lo=0.0, hi=3600.0,
                  mutability="fixed", doc="Full-disk refresh period")
DOWNLINK_BPS = param("DOWNLINK_BPS", type="f64", unit="1/s", lo=0.0, hi=1.0e9,
                     mutability="design", doc="Crosslink data rate, bits per second")
COMPUTE_W = param("COMPUTE_W", type="f32", unit="W", lo=0.0, hi=500.0,
                  mutability="design", doc="Onboard compute power draw")
N_SATS = param("N_SATS", type="u8", unit="1", lo=1, hi=32, mutability="design",
               doc="Satellites in the ring", default=4)
