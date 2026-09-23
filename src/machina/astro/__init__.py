"""
machina.astro -- orbital mechanics pack.

Modified equinoctial elements, orbit geometry and the coverage components
built on them. Importing this package declares its frames (``eci``, ``ecef``,
``lvlh``) and signals into the default signal registry; ``import machina``
never imports it, so nothing in the core depends on an orbit existing.
"""

from machina.astro import signals
from machina.astro.components.single_sat_coverage import SingleSatCoverage

__all__ = ["signals", "SingleSatCoverage"]
