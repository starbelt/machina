"""
machina.astro -- orbital mechanics pack.

Modified equinoctial elements, orbit geometry and the coverage components
built on them. Importing this package declares its frames (``eci``, ``ecef``,
``lvlh``) and signals into the default signal registry and registers its
factories: the seven ``transform.*`` (:mod:`~machina.astro.transforms`), the two
``geometry.*`` (:mod:`~machina.astro.geometry`) and ``cost.smooth_coverage``
(:mod:`~machina.astro.coverage`). ``import machina`` never imports it, so
nothing in the core depends on an orbit existing.
"""

from machina.astro import coverage, geometry, signals, transforms  # noqa: F401
from machina.astro.components.single_sat_coverage import SingleSatCoverage

__all__ = ["signals", "SingleSatCoverage"]
